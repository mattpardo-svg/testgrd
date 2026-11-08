"""nx_security_analysis.py — N-X branch currents (one pypowsybl security analysis for all
must-study contingencies) and the TATL per secured branch.

The security analysis applies each contingency to an internal working copy: the live
network's connectivity is unchanged afterwards. _get_must_study_contingencies and
_register_contingencies are also used by nx_sensitivity.
"""

import csv
import logging
import math
import re
from pathlib import Path
from typing import Optional

from data_model import AEType, AssessedElement, Contingency, LfNx, select_by_side
from milp.milp_builder import _get_monitored_branch_sides
from pre_milp import cgmes_loader, log_summary
from pre_milp.cra_preprocessing import TATL_FALLBACK_DURATION_S_DEFAULT
from pre_milp.lf_parameters import build_security_analysis_parameters
from text_encoding import TEXT_ENCODING

log = logging.getLogger(__name__)


def _get_must_study_contingencies(
    co_flat_list: list[Contingency], timestamp: str,
) -> list[Contingency]:
    """Contingencies with must_study[timestamp] True ({TSO}_CO.xml normalMustStudy) and at least one equipment (the
    security analysis returns no result for none), sorted by mRID (stable matrix_id naming and CSV row order)."""
    return sorted(
        (c for c in co_flat_list if c.must_study.get(timestamp, False) and c.equipment_mrids),
        key=lambda c: c.mrid,
    )


def _register_contingencies(container, contingencies: list[Contingency]) -> None:
    """Register each Contingency on a pypowsybl ContingencyContainer (security or sensitivity
    analysis): one equipment mRID -> single-element contingency, several (e.g. a double-circuit
    tower) -> add_multiple_elements_contingency."""
    for contingency in contingencies:
        equipment_mrids = list(contingency.equipment_mrids)
        if len(equipment_mrids) == 1:
            container.add_single_element_contingency(equipment_mrids[0], contingency.mrid)
        else:
            container.add_multiple_elements_contingency(equipment_mrids, contingency.mrid)


def _contingency_result_is_usable(
    contingency: Contingency,
    status_name: str,
    branch_ids_present: set[str],
    expected_branch_ids: set[str],
    baseline_present: Optional[set[str]] = None,
) -> tuple[bool, Optional[str]]:
    """(usable, reason) for one contingency's result. Skipped only when the status is neither CONVERGED nor NO_IMPACT
    (NO_IMPACT is a valid outcome): the load flow failed. A monitored branch that is missing from the results is no
    reason to skip the contingency (it may be islanded by it): the branch is dropped for that contingency, see
    missing_branches. branch_ids_present / expected_branch_ids / baseline_present are kept for the callers."""
    if status_name not in ("CONVERGED", "NO_IMPACT"):
        return False, f"load flow failed (non-convergent, status={status_name})"
    return True, None


def missing_branches(
    contingency: Contingency, branch_ids_present: set[str], expected_branch_ids: set[str],
    baseline_present: Optional[set[str]] = None,
) -> tuple[list[str], list[str]]:
    """(tripped, dropped) monitored branches without a result after the contingency: tripped = the contingency's own
    equipment (no flow by definition); dropped = any other (for example islanded by the contingency). A branch that the
    N-0 result (baseline_present) does not have either is neither: it is not monitored at all."""
    if baseline_present is not None:
        expected_branch_ids = expected_branch_ids & baseline_present
    own_equipment_mrids = set(contingency.equipment_mrids)
    missing = expected_branch_ids - branch_ids_present
    return sorted(missing & own_equipment_mrids), sorted(missing - own_equipment_mrids)


def _fully_disconnected(network, mrid: str) -> Optional[bool]:
    """True when every terminal of the equipment is disconnected (a branch at both ends, a switch open), False when one
    is connected, None when the network does not have it (or is no network)."""
    for accessor, flags, closed in (
        ("get_lines", ("connected1", "connected2"), True), ("get_2_windings_transformers", ("connected1", "connected2"), True),
        ("get_tie_lines", ("connected1", "connected2"), True), ("get_dangling_lines", ("connected",), True),
        ("get_generators", ("connected",), True), ("get_loads", ("connected",), True),
        ("get_shunt_compensators", ("connected",), True), ("get_static_var_compensators", ("connected",), True),
        ("get_switches", ("open",), False),
    ):
        try:
            table = getattr(network, accessor)()
            if mrid in table.index:
                row = table.loc[mrid]
                states = [bool(row[flag]) for flag in flags if flag in row]
                return not any(states) if closed else all(states)
        except Exception:
            continue
    return None


def _why_no_result(network, contingency: Contingency) -> str:
    """Why the security analysis returned nothing for a contingency. pypowsybl drops a contingency whose equipment is
    already disconnected (it has no impact); no other cause is known from the result."""
    states = [_fully_disconnected(network, mrid) for mrid in contingency.equipment_mrids]
    if states and all(state is True for state in states):
        return NO_IMPACT_REASON
    return "no post-contingency result returned (reason not given by the security analysis)"


NO_IMPACT_REASON = "no impact: equipment already disconnected"


# ---------------------------------------------------------------------------
# N-X branch currents (lf_nx)
# ---------------------------------------------------------------------------


def _describe_branch(network, branch_mrid: str) -> str:
    return cgmes_loader.describe_branch(network, branch_mrid)


def log_skipped_contingencies(skipped: dict[str, str]) -> None:
    """One warning for the skipped must-study contingencies: the count, the first 5 ids and how many per reason
    (the reason up to its first "(" or dash); every contingency with its full reason at debug level."""
    if not skipped:
        return
    # one line per reason: "no impact" is information, anything else a warning
    by_reason: dict[str, list[str]] = {}
    for mrid in sorted(skipped):
        by_reason.setdefault(re.split(r" \(| — ", skipped[mrid], maxsplit=1)[0], []).append(mrid)
    for reason, mrids in by_reason.items():
        level = logging.INFO if reason.startswith("no impact") else logging.WARNING
        # the same summary is not repeated at every refresh of the same run
        if not log_summary.first_time(("nx-skipped", reason, tuple(mrids))):
            continue
        log_summary.log_ids(log, level, "compute_lf_nx: %d must-study contingency(ies) skipped - " + reason.replace("%", "%%"), mrids)
    for mrid in sorted(skipped):
        log.debug("compute_lf_nx: contingency %s skipped — %s", mrid, skipped[mrid])


def log_dropped_branches(dropped: dict[str, list[str]]) -> None:
    """One warning per run: the contingencies that were evaluated without some monitored branches (no result after them),
    the number of (branch, contingency) pairs dropped, the first 5 contingency ids; every pair at debug level."""
    if not dropped:
        return
    pairs = sum(len(branches) for branches in dropped.values())
    if log_summary.first_time(("nx-dropped", tuple(sorted((c, tuple(b)) for c, b in dropped.items())))):
        log_summary.warn_ids(
            log, "compute_lf_nx: %d contingency(ies) evaluated without the monitored branches that have no result after them "
                 f"(network split?): {pairs} (branch, contingency) pair(s) dropped", sorted(dropped))
    for mrid in sorted(dropped):
        log.debug("compute_lf_nx: contingency %s evaluated without the branches %s", mrid, dropped[mrid])


def _index_branch_results(
    branch_results, value_columns: tuple = ("i1", "i2"),
) -> tuple[dict[str, set], dict[tuple[str, str], tuple]]:
    """One pass over the security analysis branch_results (index contingency_id,
    operator_strategy_id, branch_id): ({contingency: branch ids present},
    {(contingency, branch): raw (i1, i2) of its FIRST row}), as a per-contingency selection
    followed by .xs(branch).iloc[0] would give. Values are converted where they are read.
    value_columns ("p1", "p2") for the DC analysis: the raw values are then MW."""
    present: dict[str, set] = {}
    first_currents: dict[tuple[str, str], tuple] = {}
    contingency_ids = branch_results.index.get_level_values("contingency_id")
    branch_ids = branch_results.index.get_level_values("branch_id")
    i1_values = branch_results[value_columns[0]].to_numpy()
    i2_values = branch_results[value_columns[1]].to_numpy()
    for contingency_id, branch_id, i1, i2 in zip(contingency_ids, branch_ids, i1_values, i2_values):
        present.setdefault(contingency_id, set()).add(branch_id)
        first_currents.setdefault((contingency_id, branch_id), (i1, i2))
    return present, first_currents


def compute_lf_nx(
    network,
    co_flat_list: list[Contingency],
    ae_flat_list: list[AssessedElement],
    timestamp: str,
    lf_initial: Optional[dict[str, dict[int, float]]] = None,
    output_path: Optional[Path] = None,
    use_reactive_limits: bool = False,
    extra_lf_params: dict = None,
    dc_converter=None,
) -> tuple[dict[str, dict[str, dict[int, float]]], dict[str, str]]:
    """N-X branch currents of every must-study contingency from one security analysis run.
    Returns (lf_nx, skipped): lf_nx = {contingency: {branch: {1: A, 2: A}}} (both sides, the
    side is selected where used), skipped = {contingency: reason} (non-convergent or likely
    split). lf_initial only fills the ae_type/limit_ka/loading_pct/tso CSV columns.
    use_reactive_limits / extra_lf_params: the pipeline's load-flow settings (lf_parameters).
    dc_converter (calculation_mode dc): DC security analysis; the currents are computed from its
    p1/p2 by dc_converter (dc_conversion), not taken from its i1/i2. None = the AC analysis."""
    must_study_contingencies = _get_must_study_contingencies(co_flat_list, timestamp)
    # a must-study contingency without equipment (the parser keeps it, DECISIONS / ncp_dataset_coverage C8) is not sent
    # to the security analysis, which returns no result for it; it is listed as skipped
    no_equipment = sorted(c.mrid for c in co_flat_list if c.must_study.get(timestamp, False) and not c.equipment_mrids)
    if not must_study_contingencies and no_equipment:
        skipped_only = {mrid: "no equipment in the contingency" for mrid in no_equipment}
        log_skipped_contingencies(skipped_only)
        return {}, skipped_only
    if not must_study_contingencies:
        raise ValueError(f"No must-study contingencies found for timestamp {timestamp}")

    branch_sides = _get_monitored_branch_sides(ae_flat_list)
    if not branch_sides:
        raise ValueError("No monitored branch IDs found in AEFlatList")

    try:
        import pypowsybl as pp
    except Exception as exc:
        raise ImportError("pypowsybl is required to compute N-X load flows") from exc

    security_analysis = pp.security.create_analysis()
    _register_contingencies(security_analysis, must_study_contingencies)
    security_analysis.add_monitored_elements(branch_ids=sorted(branch_sides.keys()))

    # same load-flow parameters as the initial load flow (lf_parameters)
    security_parameters = build_security_analysis_parameters(use_reactive_limits, extra_lf_params)
    if dc_converter is None:
        result = security_analysis.run_ac(network, parameters=security_parameters)
        present_by_contingency, first_currents = _index_branch_results(result.branch_results)
    else:
        result = security_analysis.run_dc(network, parameters=security_parameters)
        present_by_contingency, first_currents = _index_branch_results(result.branch_results, ("p1", "p2"))

    lf_nx = LfNx()
    skipped_contingencies: dict[str, str] = {mrid: "no equipment in the contingency" for mrid in no_equipment}
    expected_branch_ids = set(branch_sides.keys())
    # the N-0 result (contingency id "") tells which monitored branches have a result at all (an open branch has none)
    baseline_present = present_by_contingency.get("")
    if baseline_present:
        lf_nx.not_monitored = expected_branch_ids - baseline_present
        not_monitored = sorted(lf_nx.not_monitored)
        for branch_mrid in not_monitored:
            log.debug("compute_lf_nx: monitored branch %s (%s) has no result in the N-0 analysis",
                      branch_mrid, _describe_branch(network, branch_mrid))
        if not_monitored and log_summary.first_time(("nx-branches-not-in-n0", tuple(not_monitored))):
            log_summary.warn_ids(
                log, "compute_lf_nx: %d monitored branch(es) have no result in the N-0 analysis (not connected?) - "
                "excluded from the N-X monitoring",
                [f"{branch_mrid} ({_describe_branch(network, branch_mrid)})" for branch_mrid in not_monitored])
        expected_branch_ids &= baseline_present
    else:
        baseline_present = None

    status_counts: dict[str, int] = {}
    nan_pairs: list[str] = []                  # "contingency/branch" with a NaN current in the result
    for contingency in must_study_contingencies:
        post_contingency_result = result.post_contingency_results.get(contingency.mrid)
        if post_contingency_result is None:
            reason = _why_no_result(network, contingency)
            skipped_contingencies[contingency.mrid] = reason
            status_counts["no result"] = status_counts.get("no result", 0) + 1
            continue

        status_name = str(getattr(post_contingency_result.status, "name", post_contingency_result.status))
        status_counts[status_name] = status_counts.get(status_name, 0) + 1

        branch_ids_present = present_by_contingency.get(contingency.mrid, set())

        is_usable, skip_reason = _contingency_result_is_usable(
            contingency, status_name, branch_ids_present, expected_branch_ids, baseline_present=baseline_present,
        )
        if not is_usable:
            skipped_contingencies[contingency.mrid] = skip_reason
            log.debug(
                "compute_lf_nx: contingency %s (%s) skipped — %s",
                contingency.mrid, contingency.name, skip_reason,
            )
            continue

        # A branch without a usable current is LEFT OUT of lf_nx (never given 0.0 A): the N-X constraint of that
        # (branch, contingency) pair is then not built. No result after the contingency: the contingency's own
        # equipment (no flow by definition, tripped) or another monitored branch (islanded: dropped for this
        # contingency only, one summary); a NaN current is reported where the pairs are built.
        tripped_branches, dropped_branches = missing_branches(
            contingency, branch_ids_present, expected_branch_ids, baseline_present=baseline_present,
        )
        per_branch_currents: dict[str, dict[int, float]] = {}
        for branch_mrid in sorted(expected_branch_ids):
            currents = first_currents.get((contingency.mrid, branch_mrid))
            if currents is None:
                continue
            i1, i2 = float(currents[0]), float(currents[1])
            if dc_converter is not None:
                i1, i2 = dc_converter.current(branch_mrid, 1, i1), dc_converter.current(branch_mrid, 2, i2)
            if math.isnan(i1) or math.isnan(i2):
                nan_pairs.append(f"{contingency.mrid}/{branch_mrid}")
                continue
            per_branch_currents[branch_mrid] = {1: i1, 2: i2}
        if tripped_branches:
            lf_nx.tripped[contingency.mrid] = tripped_branches
            log.debug(
                "compute_lf_nx: contingency %s: %d monitored branch(es) are its own tripped equipment "
                "- no current recorded: %s", contingency.mrid, len(tripped_branches), tripped_branches,
            )
        if dropped_branches:
            lf_nx.dropped[contingency.mrid] = dropped_branches

        lf_nx[contingency.mrid] = per_branch_currents

    log_dropped_branches(lf_nx.dropped)
    if nan_pairs and log_summary.first_time(("nx-nan-currents", tuple(nan_pairs))):
        log_summary.warn_ids(
            log, "compute_lf_nx: %d (branch, contingency) pair(s) have a NaN current in the N-X result and are left out of lf_nx", nan_pairs)
    log_skipped_contingencies(skipped_contingencies)
    if status_counts:
        log.info("compute_lf_nx: security analysis status per must-study contingency: %s",
                 ", ".join(f"{count} {name}" for name, count in sorted(status_counts.items())))
    if lf_initial is not None:
        branch_ae_types_log = {
            str(ae.branch_mrid): ae.ae_type for ae in ae_flat_list if ae.branch_mrid
        }
        secured_count = sum(1 for t in branch_ae_types_log.values() if t == AEType.SECURED)
        scanned_count = sum(1 for t in branch_ae_types_log.values() if t == AEType.SCANNED)
        log.info(
            "compute_lf_nx: %d/%d must-study contingencies usable (%d skipped) — "
            "monitored branches by ae_type: %d secured, %d scanned",
            len(lf_nx), len(must_study_contingencies), len(skipped_contingencies),
            secured_count, scanned_count,
        )
    else:
        log.info(
            "compute_lf_nx: %d/%d must-study contingencies usable (%d skipped)",
            len(lf_nx), len(must_study_contingencies), len(skipped_contingencies),
        )

    if output_path is not None:
        branch_ae_types = None
        limit_by_branch = None
        branch_tso = None
        branch_mrid_to_name = {}

        if lf_initial is not None:
            from milp.milp_builder import _get_monitored_branch_limits_and_ae_types
            from pre_milp import cgmes_loader

            limit_by_branch, branch_ae_types = _get_monitored_branch_limits_and_ae_types(
                ae_flat_list, timestamp, lf_initial,
            )
            branch_tso = {str(ae.branch_mrid): ae.tso for ae in ae_flat_list if ae.branch_mrid}
            branch_mrid_to_name = cgmes_loader.build_branch_mrid_to_name(network)

        write_lf_nx_csv(
            lf_nx, branch_sides, branch_mrid_to_name, must_study_contingencies,
            output_path, timestamp,
            branch_ae_types=branch_ae_types,
            limit_by_branch=limit_by_branch,
            branch_tso=branch_tso,
        )

    return lf_nx, skipped_contingencies


# ---------------------------------------------------------------------------
# TATL per secured branch
# ---------------------------------------------------------------------------


def compute_tatl_data(
    network,
    ae_flat_list: list[AssessedElement],
    timestamp: str,
    output_path: Optional[Path] = None,
    fallback_duration_seconds: int = TATL_FALLBACK_DURATION_S_DEFAULT,
) -> dict[str, dict]:
    """TATL per monitored SECURED branch on its AssessedElement.branch_side, from
    network.get_operational_limits(): {branch: {"tatl_limit_ampere", "tatl_duration_seconds",
    "side", "was_corrected"}}.
    - no TATL row: TATL = the branch's PATL at `timestamp` (kA -> A), duration
      fallback_duration_seconds (the duration the curative reaction times are compared with;
      such a branch gets no TATL constraint, TATL = PATL);
    - several TATLs on one side: no entry (warning; real data is not discarded);
    - TATL below PATL: raised to PATL (counted, one summary line);
    - neither TATL nor PATL: no entry (logged).
    Permanent limits have acceptable_duration -1 here (2147483647 in
    limit_violations), so `acceptable_duration > 0` selects the TATLs.
    """
    secured_branch_sides: dict[str, int] = {
        str(ae.branch_mrid): int(getattr(ae, "branch_side", 1))
        for ae in ae_flat_list
        if ae.ae_type == AEType.SECURED and ae.branch_mrid
    }
    if not secured_branch_sides:
        raise ValueError("No monitored SECURED branch IDs found in AEFlatList")

    # resolved PATL (kA) per branch at this timestamp: the source when the CGM has no TATL
    patl_ka_by_branch: dict[str, float] = {
        str(ae.branch_mrid): ae.patl_ka[timestamp]
        for ae in ae_flat_list
        if ae.ae_type == AEType.SECURED and ae.branch_mrid and timestamp in ae.patl_ka
    }

    # the limits of the branches (dangling lines on side 1, tie lines by their two dangling lines): build_branch_limits
    limits_df = cgmes_loader.build_branch_limits(network)

    limits_df_has_rows = limits_df is not None and not limits_df.empty
    if not limits_df_has_rows:
        log.info(
            "compute_tatl_data: network has no operational limits at all — "
            "every SECURED branch will fall back to its own PATL (or be "
            "absent, if PATL is also unresolved)",
        )

    side_number_to_name = {1: "ONE", 2: "TWO"}

    tatl_data: dict[str, dict] = {}
    tatl_corrected_count = 0
    own_patl_as_tatl: list[str] = []          # branches without a CGM TATL, logged as one summary
    no_tatl_no_patl: list[str] = []
    for branch_mrid, side in secured_branch_sides.items():
        side_name = side_number_to_name.get(side, "ONE")

        if limits_df_has_rows:
            branch_rows = limits_df[
                (limits_df["type"] == "CURRENT")
                & (limits_df["acceptable_duration"] > 0)
                & (limits_df["branch_id"] == branch_mrid)
                & (limits_df["side_number"] == side)
            ]
        else:
            branch_rows = None  # no operational limits data at all — same as "no rows for this branch"

        if branch_rows is None or branch_rows.empty:
            # no TATL row: TATL = the branch's own PATL
            fallback_patl_ka = patl_ka_by_branch.get(branch_mrid)
            if fallback_patl_ka is None:
                no_tatl_no_patl.append(branch_mrid)
                continue
            own_patl_as_tatl.append(branch_mrid)
            tatl_data[branch_mrid] = {
                "tatl_limit_ampere": fallback_patl_ka * 1000.0,
                "tatl_duration_seconds": int(fallback_duration_seconds),
                "side": side,
                "was_corrected": False,
            }
            continue

        if len(branch_rows) > 1:
            log.warning(
                "compute_tatl_data: branch %s side %s has %d TATL entries "
                "(exactly one expected) — skipped",
                branch_mrid, side_name, len(branch_rows),
            )
            continue

        row = branch_rows.iloc[0]
        tatl_limit_ampere = float(row["value"])

        # A TATL below the PATL is raised to the PATL (a temporary rating is never lower).
        # Counted and reported in one summary line (it can hit most branches of a dataset)
        # and marked per row in tatl_data.csv.
        fallback_patl_ka = patl_ka_by_branch.get(branch_mrid)
        was_corrected = False
        if fallback_patl_ka is not None and tatl_limit_ampere < fallback_patl_ka * 1000.0:
            tatl_limit_ampere = fallback_patl_ka * 1000.0
            was_corrected = True
            tatl_corrected_count += 1

        tatl_data[branch_mrid] = {
            "tatl_limit_ampere": tatl_limit_ampere,
            "tatl_duration_seconds": int(row["acceptable_duration"]),
            "side": side,
            "was_corrected": was_corrected,
        }

    if no_tatl_no_patl:
        log_summary.warn_ids(
            log, f"compute_tatl_data: %d branch(es) have no CGM TATL AND no resolvable PATL for timestamp {timestamp} — "
                 "no TATL constraint possible for these", no_tatl_no_patl)
    if own_patl_as_tatl:
        log_summary.log_ids(
            log, logging.INFO, f"compute_tatl_data: %d branch(es) have no CGM TATL — falling back to their own PATL as TATL, "
                               f"duration={int(fallback_duration_seconds)}s", own_patl_as_tatl)
    log.info(
        "compute_tatl_data: resolved TATL for %d/%d monitored SECURED branch(es)",
        len(tatl_data), len(secured_branch_sides),
    )
    if tatl_corrected_count:
        log.warning(
            "compute_tatl_data: %d of %d resolved TATL value(s) were LOWER than "
            "that branch's own PATL (physically backwards — TATL should never be "
            "less permissive than PATL) — corrected to TATL = PATL. See "
            "tatl_data.csv's own was_corrected column for exactly which branches.",
            tatl_corrected_count, len(tatl_data),
        )

    if output_path is not None:
        write_tatl_data_csv(tatl_data, {}, output_path)

    return tatl_data


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def write_lf_nx_csv(
    lf_nx: dict[str, dict[str, dict[int, float]]],
    branch_sides: dict[str, int],
    branch_mrid_to_name: dict[str, str],
    must_study_contingencies: list[Contingency],
    output_path: Path,
    timestamp: str,
    branch_ae_types: Optional[dict[str, "AEType"]] = None,
    limit_by_branch: Optional[dict[str, float]] = None,
    branch_tso: Optional[dict[str, str]] = None,
) -> None:
    """lf_nx_initial.csv — one row per (contingency, branch), side-aware. limit_ka is the
    limit_by_branch value (PATL for secured, effective limit for scanned) and loading_pct is
    computed against it; tso/loading_pct/limit_ka/ae_type are blank when their map is None.
    `converged` is always TRUE: compute_lf_nx leaves unusable contingencies out of lf_nx.
    """
    branch_ae_types = branch_ae_types or {}
    limit_by_branch = limit_by_branch or {}
    branch_tso = branch_tso or {}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    contingency_names = {c.mrid: c.name for c in must_study_contingencies}
    with open(output_path, "w", newline="", encoding=TEXT_ENCODING) as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "contingency_mrid", "branch_mrid", "branch_name",
            "tso", "ae_type", "side", "i_ka", "loading_pct", "limit_ka", "converged",
        ])
        for contingency_mrid in sorted(lf_nx):
            per_branch_currents = lf_nx[contingency_mrid]
            for branch_mrid in sorted(per_branch_currents):
                side = branch_sides.get(branch_mrid, 1)
                current_ampere = select_by_side(per_branch_currents[branch_mrid], side, branch_mrid)
                i_ka = float(current_ampere) / 1000.0 if current_ampere is not None else ""
                branch_name = branch_mrid_to_name.get(branch_mrid, branch_mrid)

                tso = branch_tso.get(branch_mrid, "")
                ae_type = branch_ae_types.get(branch_mrid)
                ae_type_str = ae_type.value if ae_type is not None else ""

                limit_ampere = limit_by_branch.get(branch_mrid)
                limit_ka = limit_ampere / 1000.0 if limit_ampere else ""
                loading_pct = (
                    (float(current_ampere) / limit_ampere) * 100.0
                    if (current_ampere is not None and limit_ampere) else ""
                )

                writer.writerow([
                    timestamp, contingency_mrid, branch_mrid, branch_name,
                    tso, ae_type_str, side, i_ka, loading_pct, limit_ka, "TRUE",
                ])

    log.info("Written N-X initial load flow CSV: %s", output_path)


def write_tatl_data_csv(
    tatl_data: dict[str, dict],
    branch_mrid_to_name: dict[str, str],
    output_path: Path,
) -> None:
    """tatl_data.csv — [branch_mrid, branch_name, side, tatl_limit_ka, tatl_duration_seconds,
    was_corrected], one row per branch with a TATL. was_corrected: a CGM TATL below the PATL was
    raised to the PATL (False for a TATL set from the PATL because no TATL row exists)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding=TEXT_ENCODING) as f:
        writer = csv.writer(f)
        writer.writerow(["branch_mrid", "branch_name", "side", "tatl_limit_ka", "tatl_duration_seconds", "was_corrected"])
        for branch_mrid in sorted(tatl_data):
            entry = tatl_data[branch_mrid]
            branch_name = branch_mrid_to_name.get(branch_mrid, branch_mrid)
            writer.writerow([
                branch_mrid, branch_name, entry["side"],
                float(entry["tatl_limit_ampere"]) / 1000.0, entry["tatl_duration_seconds"],
                entry.get("was_corrected", False),
            ])

    log.info("Written TATL data CSV: %s", output_path)
