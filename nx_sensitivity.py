"""nx_sensitivity.py — N-X sensitivities of PSTs (A per degree) and RD units (A per MW) per
must-study contingency, for preventive AND curative available remedial actions, with the
loading filter (Filter 1) and the magnitude pruning (Filter 2). _get_must_study_contingencies
and _register_contingencies come from nx_security_analysis, _get_monitored_branch_sides from
milp_builder.
"""

import csv
import logging
from collections import Counter
from pathlib import Path
from collections.abc import Mapping
from typing import Optional

import numpy as np

from data_model import AEType, AssessedElement, Contingency, PST, RDUnit, select_by_side
from pre_milp import ct_sensitivity
from pre_milp.lf_parameters import build_loadflow_parameters, build_sensitivity_parameters
from milp.milp_builder import _get_monitored_branch_sides
from pre_milp.nx_security_analysis import _get_must_study_contingencies, _register_contingencies
from pre_milp.sensitivity_table import SensitivityTable, table_from_matrices
from pre_milp.sensitivity_checks import DC_NX_RELATIVE_TOLERANCE, find_invalid_nx_contingencies
from text_encoding import TEXT_ENCODING

log = logging.getLogger(__name__)


def _get_limit_and_ae_type_by_branch_nx(
    ae_flat_list: list[AssessedElement], timestamp: str, lf_initial: dict[str, dict[int, float]],
) -> tuple[dict[str, float], dict[str, "AEType"]]:
    """({branch_mrid: limit_ampere}, {branch_mrid: AEType}) from milp_builder's limit resolver
    (PATL for SECURED, effective limit for SCANNED); imported lazily to avoid a circular import."""
    from milp.milp_builder import _get_monitored_branch_limits_and_ae_types

    return _get_monitored_branch_limits_and_ae_types(ae_flat_list, timestamp, lf_initial)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _get_available_pst_ids_nx(pst_list: list[PST], network=None) -> list[str]:
    """Available PST (PowerTransformer) mRIDs, preventive and curative, de-duplicated: a
    transformer with a preventive and a curative remedial action is one sensitivity variable
    (the same ID twice makes pypowsybl return a duplicate matrix row).
    With `network`, PSTs on a 3-winding transformer are left out (logged): pypowsybl's
    TRANSFORMER_PHASE variable does not support them.
    """
    candidates = sorted({str(pst.mrid) for pst in pst_list if pst.available})

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
            "Excluding %d PST(s) backed by a 3-winding transformer from N-X AC "
            "sensitivity (unsupported by pypowsybl's TRANSFORMER_PHASE "
            "variable type for 3W transformers): %s",
            len(excluded), excluded,
        )
    return filtered


def _get_available_rd_gu_ids_nx(rd_list: list[RDUnit]) -> list[str]:
    """Sorted, de-duplicated generating-unit mRIDs of the available RD units, preventive and
    curative (the sensitivity variable is the unit, shared by its remedial actions). The units of the
    countertrade RAs have a variable per RA and direction instead (compute_ct_nx)."""
    return ct_sensitivity.individual_unit_ids(rd_list)


def compute_ct_nx(
    network, contingencies: list[Contingency], branch_sides: dict[str, int], weights: dict[str, dict[str, float]],
    dc_converter=None, use_reactive_limits: bool = False, extra_lf_params: Optional[dict] = None,
    prune_limits: Optional[dict[str, float]] = None, prune_threshold_pct: float = 0.0,
) -> tuple[dict[str, dict[int, tuple]], dict[str, dict[str, float]]]:
    """The countertrade variables of every contingency: ({contingency: {side: (branches, variables, array (variables,
    branches))}}, {contingency: {branch: reference value}}). The per-unit sensitivities (INJECTION_ACTIVE_POWER, the
    parameters and conversion of the RD units) are read in batches of contingencies (ct_sensitivity.batch_size) and
    summed with the weights right away (ct_sensitivity.aggregate_units); the per-unit values of a batch are dropped
    before the next one. prune_limits / prune_threshold_pct: Filter 2 as it applies to a unit entry, before the sum."""
    import pypowsybl as pp
    blocks: dict[str, dict[int, tuple]] = {}
    references: dict[str, dict[str, float]] = {}
    if not weights or not contingencies:
        return blocks, references
    unit_ids = ct_sensitivity.countertrade_unit_ids(weights)
    step = ct_sensitivity.batch_size(len(unit_ids), len(branch_sides))
    for start in range(0, len(contingencies), step):
        batch = contingencies[start:start + step]
        result, matrix_ids_by_prefix = _run_nx_sensitivity_analysis(
            network, batch, branch_sides, [("nx_ct", unit_ids, pp.sensitivity.SensitivityVariableType.INJECTION_ACTIVE_POWER)],
            use_reactive_limits=use_reactive_limits, extra_lf_params=extra_lf_params, dc_converter=dc_converter,
        )
        matrix_ids = matrix_ids_by_prefix["nx_ct"]
        p_ref_mw = _reference_currents_nx(result, matrix_ids)
        references.update(p_ref_mw)
        for contingency in batch:
            side_blocks: dict[int, tuple] = {}
            for side, matrix_id in matrix_ids[contingency.mrid].items():
                matrix = result.get_sensitivity_matrix(matrix_id, contingency.mrid)
                if matrix is None:
                    log.warning("compute_ct_nx: no matrix for contingency %s side %d", contingency.mrid, side)
                    continue
                branches = [str(b) for b in matrix.columns]
                values = matrix.to_numpy().T                                       # (branches, units)
                if dc_converter is not None:                                       # the factor of a branch is the same for every unit
                    reference = p_ref_mw.get(contingency.mrid, {})
                    values = values * np.array([
                        dc_converter.sensitivity(b, branch_sides.get(b, 1), 1.0, reference.get(b, float("nan"))) for b in branches
                    ])[:, None]
                thresholds = None
                if prune_limits is not None and float(prune_threshold_pct) > 0.0:
                    thresholds = np.array([float(prune_threshold_pct) * float(prune_limits.get(b) or 0.0) / 100.0 for b in branches])
                variables, aggregate = ct_sensitivity.aggregate_units(values, [str(u) for u in matrix.index], weights, thresholds)
                side_blocks[side] = (branches, variables, np.ascontiguousarray(aggregate.T))
            blocks[contingency.mrid] = side_blocks
        del result
    return blocks, references


def _run_nx_sensitivity_analysis(
    network,
    must_study_contingencies: list[Contingency],
    branch_sides: dict[str, int],
    variable_sets: list[tuple[str, list[str], object]],
    use_reactive_limits: bool = False,
    extra_lf_params: Optional[dict] = None,
    dc_converter=None,
):
    """One pypowsybl AC sensitivity analysis with, per variable set (prefix, variable ids,
    variable type), one factor matrix per side for ALL registered contingencies
    (ContingencyContextType.ALL; several contingency IDs with SPECIFIC raise
    ArrayIndexOutOfBoundsException), run once with the pipeline's load-flow parameters.
    Returns (result, {prefix: {contingency: {side: matrix_id}}}), a side only when it has
    monitored branches; get_sensitivity_matrix(matrix_id, contingency) reads one contingency.
    A variable set without ids adds no matrix.
    dc_converter (calculation_mode dc): a DC analysis with BRANCH_ACTIVE_POWER_1/2 instead (the
    only functions it supports); the matrices are then in MW per unit.
    """
    import pypowsybl as pp

    side1_branch_ids = sorted(b for b, s in branch_sides.items() if s == 1)
    side2_branch_ids = sorted(b for b, s in branch_sides.items() if s == 2)

    functions = pp.sensitivity.SensitivityFunctionType
    if dc_converter is None:
        sensitivity_analysis = pp.sensitivity.create_ac_analysis()
        function_type_side1, function_type_side2 = functions.BRANCH_CURRENT_1, functions.BRANCH_CURRENT_2
    else:
        sensitivity_analysis = pp.sensitivity.create_dc_analysis()
        function_type_side1, function_type_side2 = functions.BRANCH_ACTIVE_POWER_1, functions.BRANCH_ACTIVE_POWER_2
    _register_contingencies(sensitivity_analysis, must_study_contingencies)

    matrix_ids_by_prefix: dict[str, dict[str, dict[int, str]]] = {}
    for prefix, variable_ids, sensitivity_variable_type in variable_sets:
        side_matrix_ids: dict[int, str] = {}
        for side, branch_ids, function_type in (
            (1, side1_branch_ids, function_type_side1),
            (2, side2_branch_ids, function_type_side2),
        ):
            if not branch_ids or not variable_ids:
                continue
            matrix_id = f"{prefix}_side{side}"
            sensitivity_analysis.add_factor_matrix(
                functions_ids=branch_ids,
                variables_ids=variable_ids,
                contingencies_ids=[],
                contingency_context_type=pp.sensitivity.ContingencyContextType.ALL,
                sensitivity_function_type=function_type,
                sensitivity_variable_type=sensitivity_variable_type,
                matrix_id=matrix_id,
            )
            side_matrix_ids[side] = matrix_id
        matrix_ids_by_prefix[prefix] = {
            contingency.mrid: dict(side_matrix_ids) for contingency in must_study_contingencies
        }

    result = sensitivity_analysis.run(
        network, parameters=build_sensitivity_parameters(use_reactive_limits, extra_lf_params),
    )
    return result, matrix_ids_by_prefix


def _compute_sensitivity_matrices_nx(
    network,
    must_study_contingencies: list[Contingency],
    branch_sides: dict[str, int],
    variable_ids: list[str],
    sensitivity_variable_type,
    use_reactive_limits: bool = False,
    extra_lf_params: Optional[dict] = None,
    dc_converter=None,
):
    """_run_nx_sensitivity_analysis with one variable set (matrix ids nx_side1 / nx_side2):
    returns (result, {contingency: {side: matrix_id}})."""
    result, matrix_ids_by_prefix = _run_nx_sensitivity_analysis(
        network, must_study_contingencies, branch_sides,
        [("nx", variable_ids, sensitivity_variable_type)],
        use_reactive_limits=use_reactive_limits, extra_lf_params=extra_lf_params, dc_converter=dc_converter,
    )
    return result, matrix_ids_by_prefix["nx"]


def _matrix_entries(matrix, contingency_mrid: str, function_name: str, variable_label: str) -> dict[tuple[str, str], float]:
    """{(branch, variable): value} of one sensitivity matrix (index = variables, columns =
    branches), read with numpy in column-then-row order. A duplicated label uses its FIRST
    row/column (warning per cell, as pandas .loc would return several values); NaN stays NaN."""
    matrix_variable_ids = [str(v) for v in matrix.index]
    matrix_branch_ids = [str(b) for b in matrix.columns]
    values = matrix.to_numpy()
    row_of: dict[str, int] = {}
    for position, variable_id in enumerate(matrix_variable_ids):
        row_of.setdefault(variable_id, position)
    column_of: dict[str, int] = {}
    for position, branch_id in enumerate(matrix_branch_ids):
        column_of.setdefault(branch_id, position)
    row_count = Counter(matrix_variable_ids)
    column_count = Counter(matrix_branch_ids)
    entries: dict[tuple[str, str], float] = {}
    for branch_mrid in matrix_branch_ids:
        for variable_mrid in matrix_variable_ids:
            duplicates = row_count[variable_mrid] if row_count[variable_mrid] > 1 else column_count[branch_mrid]
            if duplicates > 1:
                log.warning(
                    "%s: matrix.loc[%s, %s] returned %d value(s) instead of one (contingency %s) — "
                    "matrix.index has a duplicate entry for this %s despite the input list having "
                    "none. Using the first value; root cause not yet confirmed.",
                    function_name, variable_mrid, branch_mrid, duplicates, contingency_mrid, variable_label,
                )
            entries[(branch_mrid, variable_mrid)] = float(values[row_of[variable_mrid], column_of[branch_mrid]])
    return entries


def _merged_blocks(unit_blocks: dict, ct_blocks: dict) -> list:
    """The blocks of the unit matrices and of the countertrade variable matrices, side by side (variables of the
    units first); a side whose two matrices have other branches makes a table impossible (None)."""
    merged = []
    for side in sorted(set(unit_blocks) | set(ct_blocks)):
        if side in unit_blocks and side in ct_blocks:
            (branches, variables, values), (ct_branches, ct_variables, ct_values) = unit_blocks[side], ct_blocks[side]
            if branches != ct_branches:
                return None
            merged.append((branches, variables + ct_variables, np.concatenate([values, ct_values], axis=0)))
        else:
            merged.append(unit_blocks.get(side) or ct_blocks[side])
    return merged


def _contingency_sensitivities(
    result, contingency_mrid: str, matrix_ids: dict, function_name: str, variable_label: str,
    branch_sides: dict, p_ref_mw: dict, dc_converter, ct_blocks: Optional[dict] = None,
):
    """The sensitivities of one contingency over its side matrices: a SensitivityTable (numpy) in the AC case when
    the matrices are consistent, otherwise the {(branch, variable): value} dict of _matrix_entries. The DC conversion
    (a function per entry) always works on the dict. ct_blocks ({side: (branches, variables, array (variables,
    branches))}, from compute_ct_nx, already in A per MW): the countertrade variables, added to the same table."""
    unit_blocks, matrices = {}, []
    for side, matrix_id in matrix_ids.items():
        matrix = result.get_sensitivity_matrix(matrix_id, contingency_mrid)
        if matrix is None:
            log.warning("%s: no matrix for contingency %s side %d", function_name, contingency_mrid, side)
            continue
        matrices.append(matrix)
        unit_blocks[side] = ([str(b) for b in matrix.columns], [str(v) for v in matrix.index], matrix.to_numpy())
    ct_blocks = ct_blocks or {}
    if dc_converter is None:
        merged = _merged_blocks(unit_blocks, ct_blocks)
        table = table_from_matrices(merged) if merged is not None else None
        if table is not None:
            return table
    # pypowsybl's matrix can hold a duplicate row for one variable (cause unknown): the first value is used and a
    # warning is logged
    per_contingency: dict[tuple[str, str], float] = {}
    for matrix in matrices:
        entries = _matrix_entries(matrix, contingency_mrid, function_name, variable_label)
        if dc_converter is not None:
            entries = _dc_convert_entries(entries, contingency_mrid, branch_sides, p_ref_mw, dc_converter)
        per_contingency.update(entries)
    for branches, variables, values in ct_blocks.values():
        for column, branch in enumerate(branches):
            for row, variable in enumerate(variables):
                per_contingency[(branch, variable)] = float(values[row, column])
    return per_contingency

def _all_zero(values) -> bool:
    return values.all_zero() if isinstance(values, SensitivityTable) else all(v == 0.0 for v in values.values())


def _reference_currents_nx(result, matrix_ids_by_contingency) -> dict[str, dict[str, float]]:
    """{contingency_mrid: {branch_mrid: reference (post-contingency) current in A}} as seen by
    the sensitivity analysis itself: the value the sensitivities are linearised around. The
    guard compares it with the N-X load flow (lf_nx) of the same state. A (contingency, side)
    the analysis returned no reference for is simply absent."""
    references: dict[str, dict[str, float]] = {}
    for contingency_mrid, matrix_ids in matrix_ids_by_contingency.items():
        per_branch: dict[str, float] = {}
        for matrix_id in matrix_ids.values():
            try:
                reference_matrix = result.get_reference_matrix(matrix_id, contingency_mrid)
            except Exception:
                reference_matrix = None
            if reference_matrix is None:
                continue
            for branch_mrid in reference_matrix.columns:
                per_branch[str(branch_mrid)] = float(reference_matrix.iloc[0][branch_mrid])
        references[contingency_mrid] = per_branch
    return references


def _guard_tolerance(dc_converter) -> dict:
    """Keyword argument of the N-X guard: the DC mode compares the reference with the DC security
    analysis with DC_NX_RELATIVE_TOLERANCE (see sensitivity_checks); the AC analyses keep the default."""
    return {} if dc_converter is None else {"tolerance": DC_NX_RELATIVE_TOLERANCE}


def _dc_references_to_currents(p_ref_mw: dict, branch_sides: dict, dc_converter) -> dict[str, dict[str, float]]:
    """{contingency: {branch: reference flow (MW)}} -> reference currents (A) for the guard."""
    return {
        contingency_mrid: {
            branch_mrid: dc_converter.current(branch_mrid, branch_sides.get(branch_mrid, 1), p)
            for branch_mrid, p in per_branch.items()
        }
        for contingency_mrid, per_branch in p_ref_mw.items()
    }


def _dc_convert_entries(entries: dict, contingency_mrid: str, branch_sides: dict, p_ref_mw: dict, dc_converter) -> dict:
    """{(branch, variable): MW per unit} of one contingency -> A per unit, with the sign of the
    reference flow of that branch side and contingency (NaN without a reference)."""
    references = p_ref_mw.get(contingency_mrid, {})
    return {
        (branch_mrid, variable_mrid): dc_converter.sensitivity(
            branch_mrid, branch_sides.get(branch_mrid, 1), value, references.get(branch_mrid, float("nan")),
        )
        for (branch_mrid, variable_mrid), value in entries.items()
    }


def _apply_loading_threshold_filter_nx(
    sensitivity_nx: dict[str, dict[tuple[str, str], float]],
    lf_nx: dict[str, dict[str, dict[int, float]]],
    branch_sides: dict[str, int],
    limit_by_branch: dict[str, float],
    min_loading_threshold_pct_nx: float,
) -> dict[str, dict[tuple[str, str], float]]:
    """Filter 1 (loading): drop every (branch, RA) entry of a (branch, contingency) pair whose
    lf_nx[c][b] / limit_by_branch[b] is below min_loading_threshold_pct_nx (PATL for secured,
    effective limit for scanned branches). Applied after the analysis, which computes every
    monitored branch. A pair without a limit or without an N-X current is dropped.
    """
    def branch_passes(per_branch_currents, branch_mrid) -> bool:
        limit_ampere = limit_by_branch.get(branch_mrid)
        if not limit_ampere or limit_ampere <= 0.0:
            return False
        side = branch_sides.get(branch_mrid, 1)
        nx_current_ampere = select_by_side(
            per_branch_currents.get(branch_mrid, {}), side, branch_mrid,
        )
        if nx_current_ampere is None:
            return False
        loading_pct = (float(nx_current_ampere) / limit_ampere) * 100.0
        return not loading_pct < float(min_loading_threshold_pct_nx)

    filtered: dict[str, Mapping[tuple[str, str], float]] = {}
    for contingency_mrid, per_branch_ra in sensitivity_nx.items():
        per_branch_currents = lf_nx.get(contingency_mrid, {})
        if isinstance(per_branch_ra, SensitivityTable):
            # the decision depends on the branch only: one per row, not one per entry
            filtered[contingency_mrid] = per_branch_ra.restricted_to_branches(
                lambda branch_mrid: branch_passes(per_branch_currents, branch_mrid)
            )
            continue
        filtered_for_contingency: dict[tuple[str, str], float] = {}
        for (branch_mrid, ra_mrid), value in per_branch_ra.items():
            if branch_passes(per_branch_currents, branch_mrid):
                filtered_for_contingency[(branch_mrid, ra_mrid)] = value
        filtered[contingency_mrid] = filtered_for_contingency
    return filtered


def compute_sensitivity_pst_nx(
    network,
    co_flat_list: list[Contingency],
    pst_list: list[PST],
    ae_flat_list: list[AssessedElement],
    timestamp: str,
    lf_nx: dict[str, dict[str, dict[int, float]]],
    lf_initial: dict[str, dict[int, float]],
    min_loading_threshold_pct_nx: float = 0.0,
    output_path: Optional[Path] = None,
    use_reactive_limits: bool = False,
    extra_lf_params: Optional[dict] = None,
    reference_out: Optional[dict] = None,
    guard_out: Optional[dict] = None,
    dc_converter=None,
) -> dict[str, dict[tuple[str, str], float]]:
    """N-X PST sensitivities dI/ds per must-study contingency:
    {contingency: {(branch, pst): A per degree}} (the A/tap conversion happens in the builder).
    Variables: preventive and curative available PSTs. Runs with the parameters of compute_lf_nx.
    reference_out: filled with {contingency: {branch: reference current A}}; guard_out: filled with
    {contingency: reason} for the guard failures (before Filter 1). Filter 1 is applied here.
    """
    must_study_contingencies = _get_must_study_contingencies(co_flat_list, timestamp)
    if not must_study_contingencies:
        raise ValueError(f"No must-study contingencies found for timestamp {timestamp}")

    branch_sides = _get_monitored_branch_sides(ae_flat_list)
    if not branch_sides:
        raise ValueError("No monitored branch IDs found in AEFlatList")

    pst_ids = _get_available_pst_ids_nx(pst_list, network=network)
    if not pst_ids:
        raise ValueError("No available PSTs found in PSTList")

    try:
        import pypowsybl as pp
    except Exception as exc:
        raise ImportError("pypowsybl is required to compute sensitivities") from exc

    result, matrix_ids_by_contingency = _compute_sensitivity_matrices_nx(
        network, must_study_contingencies, branch_sides,
        pst_ids, pp.sensitivity.SensitivityVariableType.TRANSFORMER_PHASE,
        use_reactive_limits=use_reactive_limits, extra_lf_params=extra_lf_params, dc_converter=dc_converter,
    )
    return _pst_nx_from_result(
        network, result, matrix_ids_by_contingency, must_study_contingencies, branch_sides, pst_ids,
        ae_flat_list, timestamp, lf_nx, lf_initial, min_loading_threshold_pct_nx, output_path,
        reference_out, guard_out, dc_converter,
    )


def _pst_nx_from_result(
    network, result, matrix_ids_by_contingency, must_study_contingencies, branch_sides, pst_ids,
    ae_flat_list, timestamp, lf_nx, lf_initial, min_loading_threshold_pct_nx, output_path,
    reference_out, guard_out, dc_converter=None,
) -> dict[str, dict[tuple[str, str], float]]:
    """Read-back, guard, Filter 1, log and CSV of the PST N-X sensitivities of one analysis."""
    _references_nx = _reference_currents_nx(result, matrix_ids_by_contingency)
    p_ref_mw = _references_nx
    if dc_converter is not None:
        _references_nx = _dc_references_to_currents(p_ref_mw, branch_sides, dc_converter)
    if reference_out is not None:
        reference_out.clear()
        reference_out.update(_references_nx)

    sensitivity_pst_nx: dict[str, Mapping[tuple[str, str], float]] = {}
    for contingency in must_study_contingencies:
        sensitivity_pst_nx[contingency.mrid] = _contingency_sensitivities(
            result, contingency.mrid, matrix_ids_by_contingency[contingency.mrid],
            "compute_sensitivity_pst_nx", "PST", branch_sides, p_ref_mw, dc_converter,
        )

    if guard_out is not None:
        guard_out.clear()
        guard_out.update(find_invalid_nx_contingencies(
            _references_nx,
            {co for co, values in sensitivity_pst_nx.items() if values and _all_zero(values)},
            lf_nx, branch_sides, **_guard_tolerance(dc_converter),
        ))

    limit_by_branch, branch_ae_types = _get_limit_and_ae_type_by_branch_nx(
        ae_flat_list, timestamp, lf_initial,
    )
    sensitivity_pst_nx = _apply_loading_threshold_filter_nx(
        sensitivity_pst_nx, lf_nx, branch_sides, limit_by_branch, min_loading_threshold_pct_nx,
    )

    secured_branch_count = sum(1 for t in branch_ae_types.values() if t == AEType.SECURED)
    scanned_branch_count = sum(1 for t in branch_ae_types.values() if t == AEType.SCANNED)
    log.info(
        "compute_sensitivity_pst_nx: computed for %d must-study contingencies, "
        "%d PST(s), %d monitored branch(es) [%d secured, %d scanned] "
        "(post Filter 1: %d total (branch, pst) pairs)",
        len(sensitivity_pst_nx), len(pst_ids), len(branch_sides),
        secured_branch_count, scanned_branch_count,
        sum(len(v) for v in sensitivity_pst_nx.values()),
    )

    if output_path is not None:
        from pre_milp import cgmes_loader
        branch_mrid_to_name = cgmes_loader.build_branch_mrid_to_name(network)
        write_sensitivity_nx_csv(
            sensitivity_pst_nx, branch_sides, branch_mrid_to_name, output_path, timestamp,
            branch_ae_types=branch_ae_types,
        )

    return sensitivity_pst_nx


def compute_sensitivity_rd_nx(
    network,
    co_flat_list: list[Contingency],
    rd_list: list[RDUnit],
    ae_flat_list: list[AssessedElement],
    timestamp: str,
    lf_nx: dict[str, dict[str, dict[int, float]]],
    lf_initial: dict[str, dict[int, float]],
    min_loading_threshold_pct_nx: float = 0.0,
    output_path: Optional[Path] = None,
    use_reactive_limits: bool = False,
    extra_lf_params: Optional[dict] = None,
    reference_out: Optional[dict] = None,
    guard_out: Optional[dict] = None,
    dc_converter=None,
    ct_prune_limits: Optional[dict[str, float]] = None,
    ct_prune_threshold_pct: float = 0.0,
) -> dict[str, dict[tuple[str, str], float]]:
    """N-X RD sensitivities dI/dP per must-study contingency:
    {contingency: {(branch, unit): A per MW}}, for preventive and curative available units.
    Filter 1 is applied here, as in compute_sensitivity_pst_nx. The countertrade RAs have one variable per RA and
    direction (compute_ct_nx); ct_prune_limits / ct_prune_threshold_pct: Filter 2 of the units, applied before the sum."""
    must_study_contingencies = _get_must_study_contingencies(co_flat_list, timestamp)
    if not must_study_contingencies:
        raise ValueError(f"No must-study contingencies found for timestamp {timestamp}")

    branch_sides = _get_monitored_branch_sides(ae_flat_list)
    if not branch_sides:
        raise ValueError("No monitored branch IDs found in AEFlatList")

    gu_ids = _get_available_rd_gu_ids_nx(rd_list)
    weights = ct_sensitivity.ct_weights(rd_list)
    if not gu_ids and not weights:
        raise ValueError("No available RD units found in RDList")

    try:
        import pypowsybl as pp
    except Exception as exc:
        raise ImportError("pypowsybl is required to compute sensitivities") from exc

    result, matrix_ids_by_prefix = _run_nx_sensitivity_analysis(
        network, must_study_contingencies, branch_sides,
        [("nx", gu_ids, pp.sensitivity.SensitivityVariableType.INJECTION_ACTIVE_POWER)],
        use_reactive_limits=use_reactive_limits, extra_lf_params=extra_lf_params, dc_converter=dc_converter,
    )
    ct_result = compute_ct_nx(
        network, must_study_contingencies, branch_sides, weights, dc_converter, use_reactive_limits, extra_lf_params,
        ct_prune_limits, ct_prune_threshold_pct,
    )
    return _rd_nx_from_result(
        network, result, matrix_ids_by_prefix["nx"], must_study_contingencies, branch_sides, gu_ids,
        ae_flat_list, timestamp, lf_nx, lf_initial, min_loading_threshold_pct_nx, output_path,
        reference_out, guard_out, dc_converter, ct_result=ct_result,
    )


def _rd_nx_from_result(
    network, result, matrix_ids_by_contingency, must_study_contingencies, branch_sides, gu_ids,
    ae_flat_list, timestamp, lf_nx, lf_initial, min_loading_threshold_pct_nx, output_path,
    reference_out, guard_out, dc_converter=None, zero_units_out=None, ct_result=None,
) -> dict[str, dict[tuple[str, str], float]]:
    """Read-back, guard, Filter 1, log and CSV of the RD N-X sensitivities of one analysis.
    zero_units_out (HVDC): filled with {contingency: [unit]} for every unit whose raw sensitivities
    to all monitored branches are exactly zero in that contingency (taken before Filter 1).
    ct_result: (blocks, references) of compute_ct_nx: the countertrade variables, their entries join those of the
    units; a run with countertrade units only has no unit matrix and takes its references from them."""
    ct_blocks, ct_references = ct_result or ({}, {})
    _references_nx = _reference_currents_nx(result, matrix_ids_by_contingency) if gu_ids or not ct_references else ct_references
    p_ref_mw = _references_nx
    if dc_converter is not None:
        _references_nx = _dc_references_to_currents(p_ref_mw, branch_sides, dc_converter)
    if reference_out is not None:
        reference_out.clear()
        reference_out.update(_references_nx)

    sensitivity_rd_nx: dict[str, Mapping[tuple[str, str], float]] = {}
    for contingency in must_study_contingencies:
        # duplicate matrix row: the first value is used (see _contingency_sensitivities)
        sensitivity_rd_nx[contingency.mrid] = _contingency_sensitivities(
            result, contingency.mrid, matrix_ids_by_contingency[contingency.mrid],
            "compute_sensitivity_rd_nx", "generating unit", branch_sides, p_ref_mw, dc_converter,
            ct_blocks.get(contingency.mrid),
        )

    if zero_units_out is not None:
        zero_units_out.clear()
        for contingency_mrid, values in sensitivity_rd_nx.items():
            # NaN entries (the branches the contingency itself disconnects) do not count
            zero_units = sorted(
                unit for unit in gu_ids
                if any(v == 0.0 for (_b, u), v in values.items() if u == unit)
                and all(v == 0.0 for (_b, u), v in values.items() if u == unit and v == v)
            )
            if zero_units:
                zero_units_out[contingency_mrid] = zero_units

    if guard_out is not None:
        guard_out.clear()
        guard_out.update(find_invalid_nx_contingencies(
            _references_nx,
            {co for co, values in sensitivity_rd_nx.items() if values and _all_zero(values)},
            lf_nx, branch_sides, **_guard_tolerance(dc_converter),
        ))

    limit_by_branch, branch_ae_types = _get_limit_and_ae_type_by_branch_nx(
        ae_flat_list, timestamp, lf_initial,
    )
    sensitivity_rd_nx = _apply_loading_threshold_filter_nx(
        sensitivity_rd_nx, lf_nx, branch_sides, limit_by_branch, min_loading_threshold_pct_nx,
    )

    secured_branch_count = sum(1 for t in branch_ae_types.values() if t == AEType.SECURED)
    scanned_branch_count = sum(1 for t in branch_ae_types.values() if t == AEType.SCANNED)
    log.info(
        "compute_sensitivity_rd_nx: computed for %d must-study contingencies, "
        "%d generating unit(s), %d monitored branch(es) [%d secured, %d scanned] "
        "(post Filter 1: %d total (branch, gu) pairs)",
        len(sensitivity_rd_nx), len(gu_ids), len(branch_sides),
        secured_branch_count, scanned_branch_count,
        sum(len(v) for v in sensitivity_rd_nx.values()),
    )

    if output_path is not None:
        from pre_milp import cgmes_loader
        branch_mrid_to_name = cgmes_loader.build_branch_mrid_to_name(network)
        branch_tso = {str(ae.branch_mrid): ae.tso for ae in ae_flat_list if ae.branch_mrid}
        write_sensitivity_rd_nx_csv(
            sensitivity_rd_nx, branch_sides, output_path, timestamp,
            branch_mrid_to_name=branch_mrid_to_name,
            branch_tso=branch_tso,
            branch_ae_types=branch_ae_types,
        )

    return sensitivity_rd_nx

def compute_sensitivities_nx(
    network,
    co_flat_list: list[Contingency],
    pst_list: list[PST],
    rd_list: list[RDUnit],
    ae_flat_list: list[AssessedElement],
    timestamp: str,
    lf_nx: dict[str, dict[str, dict[int, float]]],
    lf_initial: dict[str, dict[int, float]],
    min_loading_threshold_pct_nx: float = 0.0,
    with_pst: bool = True,
    with_rd: bool = True,
    pst_output_path: Optional[Path] = None,
    rd_output_path: Optional[Path] = None,
    use_reactive_limits: bool = False,
    extra_lf_params: Optional[dict] = None,
    pst_guard_out: Optional[dict] = None,
    rd_guard_out: Optional[dict] = None,
    pst_reference_out: Optional[dict] = None,
    rd_reference_out: Optional[dict] = None,
    dc_converter=None,
    hvdc_ei_mrids: Optional[list[str]] = None,
    hvdc_out: Optional[dict] = None,
    ct_prune_limits: Optional[dict[str, float]] = None,
    ct_prune_threshold_pct: float = 0.0,
) -> tuple[dict, dict]:
    """compute_sensitivity_pst_nx (when with_pst) and compute_sensitivity_rd_nx (when with_rd)
    from ONE sensitivity analysis with both variable sets (same values as two analyses):
    returns (sensitivity_pst_nx, sensitivity_rd_nx), {} for a kind that is not requested.
    dc_converter (calculation_mode dc): one DC analysis; the sensitivities are converted to A per
    unit by dc_converter (dc_conversion). None = the AC analysis.
    hvdc_ei_mrids (HVDC RAs, docs/specs/hvdc_ra_specs.md H8): the EIs of the available HVDC units (preventive
    and curative) are a third variable set of the same analysis (INJECTION_ACTIVE_POWER, like RD).
    Their result goes to hvdc_out, which is cleared and filled with "sensitivity" ({contingency:
    {(branch, EI): A per MW}}, Filter 1 applied), "zero_units" ({contingency: [EI]} of units whose
    sensitivities are all exactly zero there, H8.2: the group is unavailable in it) and "guard" ({contingency:
    reason} of the reference check, only computed here when neither PST nor RD is requested; otherwise
    the PST / RD guards cover the same references). The return value is unchanged.
    ct_prune_limits / ct_prune_threshold_pct: as in compute_sensitivity_rd_nx."""
    if not with_pst and not with_rd and not hvdc_ei_mrids:
        return {}, {}
    must_study_contingencies = _get_must_study_contingencies(co_flat_list, timestamp)
    if not must_study_contingencies:
        raise ValueError(f"No must-study contingencies found for timestamp {timestamp}")
    branch_sides = _get_monitored_branch_sides(ae_flat_list)
    if not branch_sides:
        raise ValueError("No monitored branch IDs found in AEFlatList")
    pst_ids = _get_available_pst_ids_nx(pst_list, network=network) if with_pst else []
    if with_pst and not pst_ids:
        raise ValueError("No available PSTs found in PSTList")
    gu_ids = _get_available_rd_gu_ids_nx(rd_list) if with_rd else []
    weights = ct_sensitivity.ct_weights(rd_list) if with_rd else {}
    if with_rd and not gu_ids and not weights:
        raise ValueError("No available RD units found in RDList")
    try:
        import pypowsybl as pp
    except Exception as exc:
        raise ImportError("pypowsybl is required to compute sensitivities") from exc

    variable_sets = []
    if with_pst:
        variable_sets.append(("nx_pst", pst_ids, pp.sensitivity.SensitivityVariableType.TRANSFORMER_PHASE))
    if with_rd:
        variable_sets.append(("nx_rd", gu_ids, pp.sensitivity.SensitivityVariableType.INJECTION_ACTIVE_POWER))

    hvdc_ids = sorted({str(e) for e in hvdc_ei_mrids}) if hvdc_ei_mrids else []
    if hvdc_ids:
        variable_sets.append(("nx_hvdc", hvdc_ids, pp.sensitivity.SensitivityVariableType.INJECTION_ACTIVE_POWER))
    result, matrix_ids_by_prefix = _run_nx_sensitivity_analysis(
        network, must_study_contingencies, branch_sides, variable_sets,
        use_reactive_limits=use_reactive_limits, extra_lf_params=extra_lf_params, dc_converter=dc_converter,
    )
    ct_result = compute_ct_nx(
        network, must_study_contingencies, branch_sides, weights, dc_converter, use_reactive_limits, extra_lf_params,
        ct_prune_limits, ct_prune_threshold_pct,
    )
    sensitivity_pst_nx = _pst_nx_from_result(
        network, result, matrix_ids_by_prefix["nx_pst"], must_study_contingencies, branch_sides, pst_ids,
        ae_flat_list, timestamp, lf_nx, lf_initial, min_loading_threshold_pct_nx, pst_output_path,
        pst_reference_out, pst_guard_out, dc_converter,
    ) if with_pst else {}
    sensitivity_rd_nx = _rd_nx_from_result(
        network, result, matrix_ids_by_prefix["nx_rd"], must_study_contingencies, branch_sides, gu_ids,
        ae_flat_list, timestamp, lf_nx, lf_initial, min_loading_threshold_pct_nx, rd_output_path,
        rd_reference_out, rd_guard_out, dc_converter, ct_result=ct_result,
    ) if with_rd else {}
    if hvdc_ids:
        hvdc_references: dict = {}
        zero_units: dict = {}
        sensitivity_hvdc_nx = _rd_nx_from_result(
            network, result, matrix_ids_by_prefix["nx_hvdc"], must_study_contingencies, branch_sides, hvdc_ids,
            ae_flat_list, timestamp, lf_nx, lf_initial, min_loading_threshold_pct_nx, None,
            hvdc_references, None, dc_converter, zero_units_out=zero_units,
        )
        guard: dict = {}
        if not with_pst and not with_rd:
            guard = find_invalid_nx_contingencies(
                hvdc_references, set(), lf_nx, branch_sides, **_guard_tolerance(dc_converter),
            )
        if zero_units:
            log.warning(
                "HVDC sensitivities exactly zero (unit unavailable in that contingency, its group with it): %s",
                "; ".join(f"{co}: {units}" for co, units in sorted(zero_units.items())),
            )
        if hvdc_out is not None:
            hvdc_out.clear()
            hvdc_out.update(sensitivity=sensitivity_hvdc_nx, zero_units=zero_units, guard=guard)
    return sensitivity_pst_nx, sensitivity_rd_nx


# ---------------------------------------------------------------------------
# Filter 2 (sensitivity magnitude pruning)
# ---------------------------------------------------------------------------


def _prune_by_magnitude(
    sensitivity_nx: dict[str, Mapping[tuple[str, str], float]],
    limit_by_branch: dict[str, float],
    threshold_pct: float,
) -> dict[str, Mapping[tuple[str, str], float]]:
    """Filter 2 of both kinds: drop an entry if |value| < threshold_pct * limit_by_branch[branch] / 100; a branch
    without a limit (or <= 0) is dropped with all its entries. The countertrade variables are not pruned by entry: their
    units were pruned before the sum (compute_ct_nx)."""
    def threshold_ampere(branch_mrid: str) -> float:
        return float(threshold_pct) * limit_by_branch[branch_mrid] / 100.0

    def has_limit(branch_mrid: str) -> bool:
        limit_ampere = limit_by_branch.get(branch_mrid)
        return bool(limit_ampere) and limit_ampere > 0.0

    pruned: dict[str, Mapping[tuple[str, str], float]] = {}
    for contingency_mrid, per_branch_ra in sensitivity_nx.items():
        if isinstance(per_branch_ra, SensitivityTable):
            pruned[contingency_mrid] = per_branch_ra.restricted_to_branches(has_limit).without_entries_below(
                threshold_ampere, exempt=lambda variable: variable.startswith(ct_sensitivity.CT_VARIABLE_PREFIX))
            continue
        pruned_for_contingency: dict[tuple[str, str], float] = {}
        for (branch_mrid, element_mrid), value in per_branch_ra.items():
            limit_ampere = limit_by_branch.get(branch_mrid)
            if not limit_ampere or limit_ampere <= 0.0:
                continue
            if abs(value) < float(threshold_pct) * limit_ampere / 100.0 and not str(element_mrid).startswith(ct_sensitivity.CT_VARIABLE_PREFIX):
                continue
            pruned_for_contingency[(branch_mrid, element_mrid)] = value
        pruned[contingency_mrid] = pruned_for_contingency
    return pruned


def prune_sensitivity_pst_nx(
    sensitivity_pst_nx: dict[str, Mapping[tuple[str, str], float]],
    limit_by_branch: dict[str, float],
    sensitivity_threshold_pct_pst_nx: float = 0.0,
) -> dict[str, Mapping[tuple[str, str], float]]:
    """Filter 2 for PSTs: drop a (branch, contingency, pst) entry if |value| <
    sensitivity_threshold_pct_pst_nx * limit_by_branch[branch] / 100. The branch is never
    dropped (it keeps a slack-only constraint) and every contingency key stays, possibly empty.
    0.0 is a no-op."""
    return _prune_by_magnitude(sensitivity_pst_nx, limit_by_branch, sensitivity_threshold_pct_pst_nx)


def prune_sensitivity_rd_nx(
    sensitivity_rd_nx: dict[str, Mapping[tuple[str, str], float]],
    limit_by_branch: dict[str, float],
    sensitivity_threshold_pct_rd_nx: float = 0.0,
) -> dict[str, Mapping[tuple[str, str], float]]:
    """Filter 2 for RD: as prune_sensitivity_pst_nx, with sensitivity_threshold_pct_rd_nx."""
    return _prune_by_magnitude(sensitivity_rd_nx, limit_by_branch, sensitivity_threshold_pct_rd_nx)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def write_sensitivity_nx_csv(
    sensitivity_pst_nx: dict[str, dict[tuple[str, str], float]],
    branch_sides: dict[str, int],
    branch_mrid_to_name: dict[str, str],
    output_path: Path,
    timestamp: str,
    branch_ae_types: Optional[dict[str, "AEType"]] = None,
) -> None:
    """sensitivity_pst_nx.csv — one row per (contingency, branch, PST); the value column
    di_ds_ampere_per_degree holds A per degree. ae_type is blank without branch_ae_types."""
    branch_ae_types = branch_ae_types or {}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding=TEXT_ENCODING) as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "contingency_mrid", "branch_mrid", "branch_name",
            "ae_type", "pst_mrid", "side", "di_ds_ampere_per_degree",
        ])
        for contingency_mrid in sorted(sensitivity_pst_nx):
            for (branch_mrid, pst_mrid), sensitivity in sorted(sensitivity_pst_nx[contingency_mrid].items()):
                side = branch_sides.get(branch_mrid, 1)
                branch_name = branch_mrid_to_name.get(branch_mrid, branch_mrid)
                ae_type = branch_ae_types.get(branch_mrid)
                ae_type_str = ae_type.value if ae_type is not None else ""
                writer.writerow([
                    timestamp, contingency_mrid, branch_mrid, branch_name,
                    ae_type_str, pst_mrid, side, sensitivity,
                ])

    log.info("Written N-X PST sensitivity CSV: %s", output_path)


def write_sensitivity_rd_nx_csv(
    sensitivity_rd_nx: dict[str, dict[tuple[str, str], float]],
    branch_sides: dict[str, int],
    output_path: Path,
    timestamp: str,
    branch_mrid_to_name: Optional[dict[str, str]] = None,
    branch_tso: Optional[dict[str, str]] = None,
    branch_ae_types: Optional[dict[str, "AEType"]] = None,
) -> None:
    """sensitivity_rd_nx.csv — one row per (contingency, branch, generating unit), value column
    di_dp_ampere_per_mw; branch_name, tso and ae_type are blank when their maps are not given."""
    branch_mrid_to_name = branch_mrid_to_name or {}
    branch_tso = branch_tso or {}
    branch_ae_types = branch_ae_types or {}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding=TEXT_ENCODING) as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "contingency_mrid", "branch_mrid", "branch_name",
            "ae_type", "gu_mrid", "tso", "side", "di_dp_ampere_per_mw",
        ])
        for contingency_mrid in sorted(sensitivity_rd_nx):
            for (branch_mrid, gu_mrid), sensitivity in sorted(sensitivity_rd_nx[contingency_mrid].items()):
                side = branch_sides.get(branch_mrid, 1)
                branch_name = branch_mrid_to_name.get(branch_mrid, branch_mrid)
                tso = branch_tso.get(branch_mrid, "")
                ae_type = branch_ae_types.get(branch_mrid)
                ae_type_str = ae_type.value if ae_type is not None else ""
                writer.writerow([
                    timestamp, contingency_mrid, branch_mrid, branch_name,
                    ae_type_str, gu_mrid, tso, side, sensitivity,
                ])

    log.info("Written N-X RD sensitivity CSV: %s", output_path)


